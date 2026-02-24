<template>
    <section class="bg-gray-100 rounded-lg shadow-md h-32/32 flex flex-col">
        <div class="flex bg-gray-300 rounded-t-lg">
            <img v-if="iconsrc" :src="iconsrc" class="h-12 w-12 mx-auto m-2 ml-2" />
            <p class="text-2xl  text-center font-bold p-2 color-main px-4">
                <slot mdc-unwrap="p" />
            </p>
        </div>
        <div class="mx-2 flex-1 flex flex-col">
            <p class="mt-1">
                <b>Authors:</b> <span class="inline-block align-middle ml-1">
                    <slot name="authors" />
                </span>
            </p>
            <p class="mt-1">
                <b>Venue:</b> <span class="inline-block align-middle ml-1">
                    <slot name="venue" />
                </span>
            </p>
            <div class="mx-2 my-2">
                <ProgressBar :progress="progress" />
            </div>
            <hr class="border-t-2 border-gray-300 mb-4" />
            <slot name="description" />

            <div class="flex-1 flex justify-center items-center">
                <slot name="image" />
            </div>

            <div class="flex flex-row">
                <slot name="left" />
                <slot name="right" />
            </div>

        </div>
    </section>
</template>

<script lang="ts" setup>
import { defineProps } from 'vue';
import ProgressBar from './ProgressBar.vue';

const { progress, iconsrc } = defineProps<{
    progress: number;
    iconsrc: string;
}>();
</script>